#include "arm_arch.h"

unsigned int OPENSSL_armcap_P = 0;
unsigned int OPENSSL_arm_midr = 0;
unsigned int OPENSSL_armv8_rsa_neonized = 0;

#ifdef _WIN32
void OPENSSL_cpuid_setup(void)
{
    OPENSSL_armcap_P |= ARMV7_NEON;
    OPENSSL_armv8_rsa_neonized = 1;
    if (IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE)) {
