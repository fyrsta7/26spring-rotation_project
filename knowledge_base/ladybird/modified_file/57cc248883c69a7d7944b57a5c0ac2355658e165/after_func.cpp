        out = out.slice(outsize - size, size);
    }
}

void RSA::decrypt(ReadonlyBytes in, Bytes& out)
{
    auto in_integer = UnsignedBigInteger::import_data(in.data(), in.size());

    UnsignedBigInteger m;
    if (m_private_key.prime1().is_zero() || m_private_key.prime2().is_zero()) {
        m = NumberTheory::ModularPower(in_integer, m_private_key.private_exponent(), m_private_key.modulus());
    } else {
        auto m1 = NumberTheory::ModularPower(in_integer, m_private_key.exponent1(), m_private_key.prime1());
        auto m2 = NumberTheory::ModularPower(in_integer, m_private_key.exponent2(), m_private_key.prime2());
        if (m1 < m2)
            m1 = m1.plus(m_private_key.prime1());

        VERIFY(m1 >= m2);

        auto h = NumberTheory::Mod(m1.minus(m2).multiplied_by(m_private_key.coefficient()), m_private_key.prime1());
        m = m2.plus(h.multiplied_by(m_private_key.prime2()));
    }

    auto size = m.export_data(out);
    auto align = m_private_key.length();
    auto aligned_size = (size + align - 1) / align * align;

