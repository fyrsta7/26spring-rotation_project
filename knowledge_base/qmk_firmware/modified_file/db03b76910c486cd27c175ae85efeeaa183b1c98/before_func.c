  }
  last_timer = timer_read();
  g = r = b = 0;
  switch( pos ) {
    case 0: r = maxval; break;
    case 1: g = maxval; break;
    case 2: b = maxval; break;
  }
  rgblight_setrgb(r, g, b);
  pos = (pos + 1) % 3;
}
#endif

#ifdef RGBLIGHT_EFFECT_ALTERNATING
void rgblight_effect_alternating(void){
  static uint16_t last_timer = 0;
  static uint16_t pos = 0;
  if (timer_elapsed(last_timer) < 500) {
    return;
  }
