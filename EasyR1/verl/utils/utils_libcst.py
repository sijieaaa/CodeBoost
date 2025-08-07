
import libcst as cst
import random
import difflib


basebooleanop_classes = cst.BaseBooleanOp.__subclasses__()
basebooleanop_classnames = list(set([e.__name__ for e in basebooleanop_classes]))
basebooleanop_instances = {name: getattr(cst, name)() for name in basebooleanop_classnames}
print("Base Boolean Ops:", basebooleanop_classnames)


basebinaryop_classes = cst.BaseBinaryOp.__subclasses__()
basebinaryop_classnames = list(set([e.__name__ for e in basebinaryop_classes]))
basebinaryop_instances = {name: getattr(cst, name)() for name in basebinaryop_classnames}
print("Base Binary Ops:", basebinaryop_classnames)

baseunaryop_classes = cst.BaseUnaryOp.__subclasses__()
baseunaryop_classnames = list(set([e.__name__ for e in baseunaryop_classes]))
baseunaryop_instances = {name: getattr(cst, name)() for name in baseunaryop_classnames}
print("Base Unary Ops:", baseunaryop_classnames)

baseaugop_classes = cst.BaseAugOp.__subclasses__()
baseaugop_classnames = list(set([e.__name__ for e in baseaugop_classes]))
baseaugop_instances = {name: getattr(cst, name)() for name in baseaugop_classnames}
print("Base Augmented Assign Ops:", baseaugop_classnames)

basecompop_classes = cst.BaseCompOp.__subclasses__()
basecompop_classnames = list(set([e.__name__ for e in basecompop_classes]))
basecompop_instances = {name: getattr(cst, name)() for name in basecompop_classnames}
print("Base Comparison Ops:", basecompop_classnames)




class ReplaceBinaryOp(cst.CSTTransformer):
    def __init__(self, prob):
        self.is_replaced = False
        self.prob = prob
    def leave_BinaryOperation(self, original_node: cst.BinaryOperation, updated_node: cst.BinaryOperation):
        if not self.is_replaced and random.random() < self.prob:
            self.is_replaced = True
            operator_name = original_node.operator.__class__.__name__
            new_operator = random.choice([op for name, op in basebinaryop_instances.items() if name != operator_name])
            return updated_node.with_changes(operator=new_operator)
        return updated_node
    
class ReplaceBooleanOp(cst.CSTTransformer):
    def __init__(self, prob):
        self.is_replaced = False
        self.prob = prob
    def leave_BooleanOperation(self, original_node: cst.BooleanOperation, updated_node: cst.BooleanOperation):
        if not self.is_replaced and random.random() < self.prob:
            self.is_replaced = True
            operator_name = original_node.operator.__class__.__name__
            new_operator = random.choice([op for name, op in basebooleanop_instances.items() if name != operator_name])
            return updated_node.with_changes(operator=new_operator)
        return updated_node
    
class ReplaceUnaryOp(cst.CSTTransformer):
    def __init__(self, prob):
        self.is_replaced = False
        self.prob = prob
    def leave_UnaryOperation(self, original_node: cst.UnaryOperation, updated_node: cst.UnaryOperation):
        if not self.is_replaced and random.random() < self.prob:
            self.is_replaced = True
            operator_name = original_node.operator.__class__.__name__
            new_operator = random.choice([op for name, op in baseunaryop_instances.items() if name != operator_name])
            return updated_node.with_changes(operator=new_operator)
        return updated_node

class ReplaceAugmentedOp(cst.CSTTransformer):
    def __init__(self, prob):
        self.is_replaced = False
        self.prob = prob
    def leave_AugAssign(self, original_node: cst.AugAssign, updated_node: cst.AugAssign):
        if not self.is_replaced and random.random() < self.prob:
            self.is_replaced = True
            operator_name = original_node.operator.__class__.__name__
            new_operator = random.choice([op for name, op in baseaugop_instances.items() if name != operator_name])
            return updated_node.with_changes(operator=new_operator)
        return updated_node


class ReplaceComparisonOp(cst.CSTTransformer):
    def __init__(self, prob):
        self.is_replaced = False
        self.prob = prob

    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison):
        # 避免修改 `__name__ == "__main__"`
        if (
            isinstance(original_node.left, cst.Name)
            and original_node.left.value == "__name__"
            and len(original_node.comparisons) == 1
            and isinstance(original_node.comparisons[0].comparator, cst.SimpleString)
            and original_node.comparisons[0].comparator.value.strip('"').strip("'") == "__main__"
        ):
            return updated_node

        if not self.is_replaced and random.random() < self.prob:
            self.is_replaced = True
            operator_name = original_node.comparisons[0].operator.__class__.__name__
            new_operator = random.choice([
                op for name, op in basecompop_instances.items() if name != operator_name
            ])
            new_comparisons = [
                cst.ComparisonTarget(operator=new_operator, comparator=updated_node.comparisons[0].comparator)
            ]
            return updated_node.with_changes(comparisons=new_comparisons)

        return updated_node



class ReplaceIfConditionWithNegation(cst.CSTTransformer):
    def __init__(self, prob):
        self.is_replaced = False
        self.prob = prob
    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        # 判断是否为 `if __name__ == "__main__"`
        test = updated_node.test
        if (
            isinstance(test, cst.Comparison)
            and isinstance(test.left, cst.Name)
            and test.left.value == "__name__"
            and len(test.comparisons) == 1
            and isinstance(test.comparisons[0].comparator, cst.SimpleString)
            and test.comparisons[0].comparator.value.strip('"').strip("'") == "__main__"
        ):
            return updated_node  # 不修改该 if
        # 正常逻辑：按概率进行替换
        if not self.is_replaced and random.random() < self.prob:
            self.is_replaced = True
            return updated_node.with_changes(
                test=cst.UnaryOperation(
                    operator=cst.Not(),
                    expression=updated_node.test
                )
            )

        return updated_node




if __name__ == "__main__":

    code_str = """
class Calculator:
    def compute(self, x, y):
        x += y
        x -= y
        x *= y
        x //= y
        x %= y
        x **= y
        x &= y
        x |= y
        return x

def compare(a, b):
    if a > b and a != b:
        return a
    elif a == b or a is b:
        return b
    elif a not in [1, 2] or b is not None:
        return 0
    return -1

def logic(x):
    return not x or x and ~x

def calc():
    return (3 + 4) * 5 - 2 / 1 ** 2 % 3
if __name__ == "__main__":
    result = compare(10, 20)
    final = calc()
    flag = -result
"""



    module = cst.parse_module(code_str)

    # 顺序使用多个 Transformer
    for ReplaceClass in [
        ReplaceBinaryOp,
        ReplaceBooleanOp,
        ReplaceUnaryOp,
        ReplaceAugmentedOp,
        ReplaceComparisonOp,
        ReplaceIfConditionWithNegation
    ]:
        module = module.visit(ReplaceClass(prob=0.9))

    code_augmented = module.code
    diff = difflib.unified_diff(
        code_str.splitlines(),
        code_augmented.splitlines(),
    )
    print("\n".join(diff))

