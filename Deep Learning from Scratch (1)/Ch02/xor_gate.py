import nand_gate as ng
import or_gate as og
import and_gate as ag

def XOR(x1, x2):
    s1 = ng.NAND(x1, x2)
    s2 = og.OR(x1, x2)
    y = ag.AND(s1, s2)
    return y

if __name__ == '__main__':
    print(XOR(0, 0))
    print(XOR(1, 0))
    print(XOR(0, 1))
    print(XOR(1, 1))