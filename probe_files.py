# probe_files.py —— 查一下本地数据里有没有 GPS 文件（给镜像方向做最终裁决）
#
# 背景：probe_mirror.py 和 probe_angle2.py 两次验证都没能确认简单镜像。
#   三种估计器给出的镜像轴分别是 87.9°（E1 最近单点）、92.8°（E3 窗口 2 m）、
#   94.3°（E3 窗口 5 m），都离 LiDAR 窗口中心 75° 很远，而且越“准”的
#   估计器离得越远；关系本身其实挺紧的（窗口 5 m 的版本相关系数 0.936，
#   已经过了 0.9 的线），问题只在轴。
#   这些估计器量的都是车的“最近边”不是车中心，有系统偏差的可能，
#   所以还不能就此判死刑。唯一能一锤定音的是 GPS 真值：
#   用 GPS 算出车辆相对基站的真实角度，直接回归“波束→角度”，轴就钉死了。
#   另外即使将来用 GPS 钉死了轴，镜像也要改成带平移的版本，不再是简单反序。
# 本脚本只做一件事：把 development_dataset 的文件夹结构列出来，
#   看有没有 GPS 文件、叫什么名字。跑完把截图（+ probe_files_results.txt）发回来：
#   有 GPS → 写 GPS 标定脚本；没有 → 主选 (a) 作废，转备选 (e) 分层分类，不再试别的估计器。
#
# 用法：python probe_files.py（默认查 C:\isac_project\LiDAR\Scenario8\development_dataset）
#   只列文件夹和文件名，不读数据，几秒钟就跑完。

import os
import sys

BASE = r"C:\isac_project"
DEFAULT_DIR = os.path.join(BASE, r"LiDAR\Scenario8\development_dataset")
OUT_NAME = "probe_files_results.txt"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)
    out(f"查看目录：{root}")
    if not os.path.isdir(root):
        out("目录不存在。")
        return
    total_files = 0
    total_bytes = 0
    gps_hits = []
    folders = {}
    examples = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        folders[rel] = len(filenames)
        examples[rel] = sorted(filenames)[:3]
        for name in filenames:
            total_files += 1
            try:
                total_bytes += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
            if "gps" in name.lower():
                gps_hits.append(os.path.join(rel, name))
    out(f"文件总数：{total_files}，总大小：{total_bytes / 1024 / 1024:.1f} MB")
    out()
    out("=== 各文件夹文件数 ===")
    for rel in sorted(folders):
        out(f"  {rel}: {folders[rel]} 个文件")
    out()
    out("=== 含 gps 的文件名（不区分大小写）===")
    if gps_hits:
        for h in gps_hits[:50]:
            out(f"  {h}")
        if len(gps_hits) > 50:
            out(f"  ……共 {len(gps_hits)} 个，只列前 50 个")
    else:
        out("  一个也没有。")
    out()
    out("=== 各文件夹的前 3 个文件名（看命名规律）===")
    for rel in sorted(examples):
        if examples[rel]:
            out(f"  {rel}: {', '.join(examples[rel])}")
    out()
    parent = os.path.dirname(root.rstrip(os.sep))
    out(f"=== 上一级目录 {parent} 下的内容 ===")
    try:
        for name in sorted(os.listdir(parent)):
            out(f"  {name}")
    except OSError as e:
        out(f"  列不出来：{e}")
    out_path = os.path.join(BASE, OUT_NAME) if os.path.isdir(BASE) else os.path.join(root, OUT_NAME)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"结果已写入：{out_path}")


if __name__ == "__main__":
    main()
