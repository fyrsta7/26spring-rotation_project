               b.num * (int64_t) c.num,
               b.den * (int64_t) c.den, INT_MAX);
    return b;
}

AVRational av_div_q(AVRational b, AVRational c)
{
    return av_mul_q(b, (AVRational) { c.den, c.num });
}

AVRational av_add_q(AVRational b, AVRational c) {
    av_reduce(&b.num, &b.den,
               b.num * (int64_t) c.den +
               c.num * (int64_t) b.den,
               b.den * (int64_t) c.den, INT_MAX);
    return b;
}

AVRational av_sub_q(AVRational b, AVRational c)
