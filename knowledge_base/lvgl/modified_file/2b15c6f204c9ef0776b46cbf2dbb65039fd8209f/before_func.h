#define LV_FONT_H

#ifdef __cplusplus
extern "C" {
#endif


/*********************
 *      INCLUDES
 *********************/
#ifdef LV_CONF_INCLUDE_SIMPLE
#include "lv_conf.h"
#else
#include "../../lv_conf.h"
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "lv_symbol_def.h"

/*********************
 *      DEFINES
 *********************/

/**********************
 *      TYPEDEFS
 **********************/

typedef struct
{
    uint32_t w_px         :8;
    uint32_t glyph_index  :24;
} lv_font_glyph_dsc_t;

typedef struct
{
    uint32_t unicode         :21;
    uint32_t glyph_dsc_index :11;
} lv_font_unicode_map_t;

typedef struct _lv_font_struct
{
    uint32_t unicode_first;
    uint32_t unicode_last;
    uint8_t h_px;
    const uint8_t * glyph_bitmap;
    const lv_font_glyph_dsc_t * glyph_dsc;
    const uint32_t * unicode_list;
    const uint8_t * (*get_bitmap)(const struct _lv_font_struct *,uint32_t);     /*Get a glyph's  bitmap from a font*/
    int16_t (*get_width)(const struct _lv_font_struct *,uint32_t);        /*Get a glyph's with with a given font*/
    struct _lv_font_struct * next_page;    /*Pointer to a font extension*/
    uint32_t bpp   		:4;                /*Bit per pixel: 1, 2 or 4*/
    uint32_t monospace	:8;				   /*Fix width (0: normal width)*/
} lv_font_t;

/**********************
 * GLOBAL PROTOTYPES
 **********************/

/**
 * Initialize the fonts
 */
void lv_font_init(void);

/**
 * Create a pair from font name and font dsc. get function. After it 'font_get' can be used for this font
 * @param child pointer to a font to join to the 'parent'
 * @param parent pointer to a font. 'child' will be joined here
 */
void lv_font_add(lv_font_t *child, lv_font_t *parent);

/**
