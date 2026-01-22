    EXR_ZIP1  = 2,
    EXR_ZIP16 = 3,
    EXR_PIZ   = 4,
    EXR_B44   = 6
};

typedef struct EXRContext {
    AVFrame picture;
    int compr;
    int bits_per_color_id;
