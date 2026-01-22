    ColumnVector(std::initializer_list<T> il) : data{il} {}

public:
    bool isNumeric() const override { return is_arithmetic_v<T>; }
