# 检查 ONNX 的量化形态：Q/DQ 节点数、权重数据类型
# 用法: python check_qat_onnx.py <model.onnx> [<model2.onnx> ...]

import sys
from collections import Counter

import onnx
from onnx import helper


def describe(path):
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    ops = Counter(n.op_type for n in g.node)
    dtype = Counter(helper.tensor_dtype_to_string(t.data_type) for t in g.initializer)
    q, dq = ops.get("QuantizeLinear", 0), ops.get("DequantizeLinear", 0)

    print(path.split("/")[-1])
    print(f"   节点总数  {len(g.node):6d}")
    print(f"   QuantizeLinear   {q:5d}")
    print(f"   DequantizeLinear {dq:5d}")
    print(f"   权重 dtype {dict(dtype)}")
    print(f"   输入 {[i.name for i in g.input]} -> 输出 {[o.name for o in g.output]}")
    # 动态范围量化(QDQ) 判定
    print(f"   => {'✅ 带 INT8 Q/DQ (QDQ 量化)' if q and dq else '❌ 无 Q/DQ，是浮点图'}")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        describe(p)
