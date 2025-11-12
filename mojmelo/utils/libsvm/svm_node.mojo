@fieldwise_init
@register_passable("trivial")
struct svm_node(Copyable, Movable):
    var index: Int
    var value: Float64
