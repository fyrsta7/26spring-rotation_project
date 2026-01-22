void RichTextLabel::_process_line(ItemFrame *p_frame,const Vector2& p_ofs,int &y, int p_width, int p_line, ProcessMode p_mode,const Ref<Font> &p_base_font,const Color &p_base_color,const Point2i& p_click_pos,Item **r_click_item,int *r_click_char,bool *r_outside,int p_char_count) {

	RID ci;
	if (r_outside)
		*r_outside=false;
	if (p_mode==PROCESS_DRAW) {
		ci=get_canvas_item();

		if (r_click_item)
			*r_click_item=NULL;

	}
	Line &l = p_frame->lines[p_line];
	Item *it = l.from;


	int line_ofs=0;
	int margin=_find_margin(it,p_base_font);
	Align align=_find_align(it);;
	int line=0;
	int spaces=0;


	if (p_mode!=PROCESS_CACHE) {

		ERR_FAIL_INDEX(line,l.offset_caches.size());
		line_ofs = l.offset_caches[line];
	}

	if (p_mode==PROCESS_CACHE) {
		l.offset_caches.clear();
		l.height_caches.clear();
		l.char_count=0;
		l.minimum_width=0;
	}

	int wofs=margin;
	int spaces_size=0;

	if (p_mode!=PROCESS_CACHE && align!=ALIGN_FILL)
		wofs+=line_ofs;

	int begin=wofs;

	Ref<Font> cfont = _find_font(it);
	if (cfont.is_null())
		cfont=p_base_font;

	//line height should be the font height for the first time, this ensures that an empty line will never have zero height and succesive newlines are displayed
	int line_height=cfont->get_height();

	Variant meta;

#define NEW_LINE \
{\
	if (p_mode!=PROCESS_CACHE) {\
		line++;\
		if (line < l.offset_caches.size())\
			line_ofs=l.offset_caches[line];\
		wofs=margin;\
		if (align!=ALIGN_FILL)\
			wofs+=line_ofs;\
	} else {\
		int used=wofs-margin;\
		switch(align) {\
			case ALIGN_LEFT: l.offset_caches.push_back(0); break;\
			case ALIGN_CENTER: l.offset_caches.push_back(((p_width-margin)-used)/2); break;\
			case ALIGN_RIGHT: l.offset_caches.push_back(((p_width-margin)-used)); break;\
			case ALIGN_FILL: l.offset_caches.push_back((p_width-margin)-used+spaces_size); break;\
		}\
		l.height_caches.push_back(line_height);\
		l.space_caches.push_back(spaces);\
	}\
	y+=line_height+get_constant(SceneStringNames::get_singleton()->line_separation);\
	line_height=0;\
	spaces=0;\
	spaces_size=0;\
	wofs=begin;\
	if (p_mode!=PROCESS_CACHE) {\
		lh=line<l.height_caches.size()?l.height_caches[line]:1;\
	}\
	if (p_mode==PROCESS_POINTER && r_click_item && p_click_pos.y>=p_ofs.y+y && p_click_pos.y<=p_ofs.y+y+lh && p_click_pos.x<p_ofs.x+wofs) {\
		if (r_outside) *r_outside=true;\
		*r_click_item=it;\
		*r_click_char=rchar;\
		return;\
	}\
}


#define ENSURE_WIDTH(m_width) \
	if (p_mode==PROCESS_CACHE) { \
		l.minimum_width=MAX(l.minimum_width,wofs+m_width);\
	}\
	if (wofs + m_width > p_width) {\
		if (p_mode==PROCESS_CACHE) {\
			if (spaces>0)	\
				spaces-=1;\
		}\
		if (p_mode==PROCESS_POINTER && r_click_item && p_click_pos.y>=p_ofs.y+y && p_click_pos.y<=p_ofs.y+y+lh && p_click_pos.x>p_ofs.x+wofs) {\
			if (r_outside) *r_outside=true;	\
			*r_click_item=it;\
			*r_click_char=rchar;\
			return;\
		}\
		NEW_LINE\
	}


#define ADVANCE(m_width) \
{\
	if (p_mode==PROCESS_POINTER && r_click_item && p_click_pos.y>=p_ofs.y+y && p_click_pos.y<=p_ofs.y+y+lh && p_click_pos.x>=p_ofs.x+wofs && p_click_pos.x<p_ofs.x+wofs+m_width) {\
		if (r_outside) *r_outside=false;	\
		*r_click_item=it;\
		*r_click_char=rchar;\
		return;\
	}\
	wofs+=m_width;\
}

#define CHECK_HEIGHT( m_height ) \
if (m_height > line_height) {\
	line_height=m_height;\
}

	Color selection_fg;
	Color selection_bg;

	if (p_mode==PROCESS_DRAW) {


		selection_fg = get_color("font_color_selected");
		selection_bg = get_color("selection_color");
	}
	int rchar=0;
	int lh=0;

	while (it) {

		switch(it->type) {

			case ITEM_TEXT: {

				ItemText *text = static_cast<ItemText*>(it);

				Ref<Font> font=_find_font(it);
				if (font.is_null())
					font=p_base_font;

				const CharType *c = text->text.c_str();				
				const CharType *cf=c;
				int fh=font->get_height();
				int ascent = font->get_ascent();
				Color color;
				bool underline=false;

				if (p_mode==PROCESS_DRAW) {
					color=_find_color(text,p_base_color);
					underline=_find_underline(text);
					if (_find_meta(text,&meta)) {

						underline=true;
					}


				} else if (p_mode==PROCESS_CACHE) {
					l.char_count+=text->text.length();

				}

				rchar=0;

				while(*c) {

					int end=0;
					int w=0;
					int fw=0;

					lh=0;
					if (p_mode!=PROCESS_CACHE) {
						lh=line<l.height_caches.size()?l.height_caches[line]:1;
					}
					bool found_space=false;

					while (c[end]!=0 && !(end && c[end-1]==' ' && c[end]!=' ')) {

						int cw = font->get_char_size(c[end],c[end+1]).width;
						if (c[end]=='\t') {
							cw=tab_size*font->get_char_size(' ').width;
						}
						w+=cw;

						if (c[end]==' ') {

							if (p_mode==PROCESS_CACHE) {
								fw+=cw;
							} else if (align==ALIGN_FILL && line<l.space_caches.size() && l.space_caches[line]>0) {
								//print_line(String(c,end)+": "+itos(l.offset_caches[line])+"/"+itos(l.space_caches[line]));
								//sub_space=cw;
								found_space=true;
							} else {
								fw+=cw;
							}
						} else {
							fw+=cw;
						}

						end++;						
					}


					ENSURE_WIDTH(w);


					//print_line("END: "+String::chr(c[end])+".");
					if (end && c[end-1]==' ') {
						spaces++;
						if (p_mode==PROCESS_CACHE) {
							spaces_size+=font->get_char_size(' ').width;
						}

						if (found_space) {
							int ln = MIN(l.offset_caches.size()-1,line);

							fw+=l.offset_caches[ln]/l.space_caches[ln];
						}

					}


					{


						int ofs=0;

						for(int i=0;i<end;i++) {
							int pofs=wofs+ofs;




							if (p_mode==PROCESS_POINTER && r_click_char && p_click_pos.y>=p_ofs.y+y && p_click_pos.y<=p_ofs.y+y+lh) {
								//int o = (wofs+w)-p_click_pos.x;


								int cw=font->get_char_size(c[i],c[i+1]).x;
								if (c[i]=='\t') {
									cw=tab_size*font->get_char_size(' ').width;
								}

								if (p_click_pos.x-cw/2>p_ofs.x+pofs) {

									rchar=int((&c[i])-cf);
									//print_line("GOT: "+itos(rchar));


									//if (i==end-1 && p_click_pos.x+cw/2 > pofs)
									//	rchar++;
									//int o = (wofs+w)-p_click_pos.x;

								//	if (o>cw/2)
								//		rchar++;
								}


								ofs+=cw;
							} else if (p_mode==PROCESS_DRAW) {

								bool selected=false;
								if (selection.active) {

									int cofs = (&c[i])-cf;
									if ((text->index > selection.from->index || (text->index == selection.from->index && cofs >=selection.from_char)) && (text->index < selection.to->index || (text->index == selection.to->index && cofs <=selection.to_char))) {
										selected=true;
									}
								}

								int cw=0;

								bool visible = visible_characters<0 || p_char_count<visible_characters;

								if (selected) {

									cw = font->get_char_size(c[i],c[i+1]).x;
									draw_rect(Rect2(p_ofs.x+pofs,p_ofs.y+y,cw,lh),selection_bg);
									if (visible)
										font->draw_char(ci,p_ofs+Point2(pofs,y+lh-(fh-ascent)),c[i],c[i+1],selection_fg);

								} else {
									if (visible)
										cw=font->draw_char(ci,p_ofs+Point2(pofs,y+lh-(fh-ascent)),c[i],c[i+1],color);
								}

								p_char_count++;
								if (c[i]=='\t') {
									cw=tab_size*font->get_char_size(' ').width;
								}


								//print_line("draw char: "+String::chr(c[i]));

								if (underline) {
									Color uc=color;
									uc.a*=0.5;
									//VS::get_singleton()->canvas_item_add_line(ci,Point2(pofs,y+ascent+2),Point2(pofs+cw,y+ascent+2),uc);
									int uy = y+lh-fh+ascent+2;
									VS::get_singleton()->canvas_item_add_line(ci,p_ofs+Point2(pofs,uy),p_ofs+Point2(pofs+cw,uy),uc);
								}
								ofs+=cw;
							}

						}
					}


					ADVANCE(fw);
					CHECK_HEIGHT(fh); //must be done somewhere
					c=&c[end];
				}


			} break;
			case ITEM_IMAGE: {

				lh=0;
				if (p_mode!=PROCESS_CACHE)
					lh = line<l.height_caches.size()?l.height_caches[line]:1;
				else
					l.char_count+=1; //images count as chars too

				ItemImage *img = static_cast<ItemImage*>(it);

				Ref<Font> font=_find_font(it);
				if (font.is_null())
					font=p_base_font;

				if (p_mode==PROCESS_POINTER && r_click_char)
					*r_click_char=0;

				ENSURE_WIDTH( img->image->get_width() );

				bool visible = visible_characters<0 || p_char_count<visible_characters;

				if (p_mode==PROCESS_DRAW && visible) {
					img->image->draw(ci,p_ofs+Point2(wofs,y+lh-font->get_descent()-img->image->get_height()));
				}
				p_char_count++;

				ADVANCE( img->image->get_width() );
				CHECK_HEIGHT( (img->image->get_height()+font->get_descent()) );

			} break;
			case ITEM_NEWLINE: {


				lh=0;
				if (p_mode!=PROCESS_CACHE)
					lh = line<l.height_caches.size()?l.height_caches[line]:1;


#if 0
				if (p_mode==PROCESS_POINTER && r_click_item ) {
					//previous last "wrapped" line
					int pl = line-1;
					if (pl<0 || lines[pl].height_caches.size()==0)
						break;
					int py=lines[pl].offset_caches[ lines[pl].offset_caches.size() -1 ];
					int ph=lines[pl].height_caches[ lines[pl].height_caches.size() -1 ];
					print_line("py: "+itos(py));
					print_line("ph: "+itos(ph));

					rchar=0;
					if (p_click_pos.y>=py && p_click_pos.y<=py+ph) {
						if (r_outside) *r_outside=true;
						*r_click_item=it;
						*r_click_char=rchar;
						return;
					}
				}

#endif
			} break;
			case ITEM_TABLE: {

				lh=0;
				ItemTable *table = static_cast<ItemTable*>(it);
				int hseparation=get_constant("table_hseparation");
				int vseparation=get_constant("table_vseparation");
				Color ccolor = _find_color(table,p_base_color);
				Vector2 draw_ofs = Point2(wofs,y);

				if (p_mode==PROCESS_CACHE) {

					int idx=0;
					//set minimums to zero
					for(int i=0;i<table->columns.size();i++) {
						table->columns[i].min_width=0;
						table->columns[i].width=0;
					}
					//compute minimum width for each cell
					for (List<Item*>::Element *E=table->subitems.front();E;E=E->next()) {
						ERR_CONTINUE(E->get()->type!=ITEM_FRAME); //children should all be frames
						ItemFrame *frame = static_cast<ItemFrame*>(E->get());

						int column = idx % table->columns.size();

						int ly=0;


						for(int i=0;i<frame->lines.size();i++) {

							_process_line(frame,Point2(),ly,p_width,i,PROCESS_CACHE,cfont,Color());
							table->columns[column].min_width=MAX( table->columns[i].min_width, frame->lines[i].minimum_width );
						}
						idx++;
					}

					//compute available width and total radio (for expanders)


					int total_ratio=0;
					int available_width=p_width - hseparation * (table->columns.size() -1);
					table->total_width=hseparation;

					for(int i=0;i<table->columns.size();i++) {
						available_width-=table->columns[i].min_width;
						if (table->columns[i].expand)
							total_ratio+=table->columns[i].expand_ratio;
					}

					//assign actual widths

					for(int i=0;i<table->columns.size();i++) {
						table->columns[i].width = table->columns[i].min_width;
						if (table->columns[i].expand)
							table->columns[i].width+=table->columns[i].expand_ratio*available_width/total_ratio;
						table->total_width+=table->columns[i].width+hseparation;
					}

					//compute caches properly again with the right width
					idx=0;
					for (List<Item*>::Element *E=table->subitems.front();E;E=E->next()) {
						ERR_CONTINUE(E->get()->type!=ITEM_FRAME); //children should all be frames
						ItemFrame *frame = static_cast<ItemFrame*>(E->get());

						int column = idx % table->columns.size();


						for(int i=0;i<frame->lines.size();i++) {

							int ly=0;
							_process_line(frame,Point2(),ly,table->columns[column].width,i,PROCESS_CACHE,cfont,Color());
							frame->lines[i].height_cache=ly; //actual height
							frame->lines[i].height_accum_cache=ly; //actual height
						}
						idx++;
					}

				}



				Point2 offset(hseparation,vseparation);

				int row_height=0;
				//draw using computed caches
				int idx=0;
				for (List<Item*>::Element *E=table->subitems.front();E;E=E->next()) {
					ERR_CONTINUE(E->get()->type!=ITEM_FRAME); //children should all be frames
					ItemFrame *frame = static_cast<ItemFrame*>(E->get());

					int column = idx % table->columns.size();

					int ly=0;
					int yofs=0;


					for(int i=0;i<frame->lines.size();i++) {

						if (p_mode==PROCESS_DRAW) {
							_process_line(frame,p_ofs+offset+draw_ofs+Vector2(0,yofs),ly,table->columns[column].width,i,PROCESS_DRAW,cfont,ccolor);
						} else if (p_mode==PROCESS_POINTER) {
							_process_line(frame,p_ofs+offset+draw_ofs+Vector2(0,yofs),ly,table->columns[column].width,i,PROCESS_POINTER,cfont,ccolor,p_click_pos,r_click_item,r_click_char,r_outside);
						}
						yofs+=frame->lines[i].height_cache;
						if (p_mode==PROCESS_CACHE) {
							frame->lines[i].height_accum_cache=offset.y+draw_ofs.y+frame->lines[i].height_cache;
						}

					}

					row_height=MAX(yofs,row_height);
					offset.x+=table->columns[column].width+hseparation;

					if (column==table->columns.size()-1) {

						offset.y+=row_height+vseparation;
						offset.x=hseparation;
						row_height=0;
					}
					idx++;
				}

				int total_height = offset.y;
				if (row_height) {
					total_height=row_height+vseparation;
				}



				ADVANCE( table->total_width );
				CHECK_HEIGHT( total_height );

			} break;

			default: {}

		}


		Item *itp = it;

		it = _get_next_item(it);

		if (p_mode == PROCESS_POINTER && r_click_item && itp && !it && p_click_pos.y>p_ofs.y+y+lh) {
			//at the end of all, return this
			if (r_outside) *r_outside=true;
			*r_click_item=itp;
			*r_click_char=rchar;
			return;
		}

		if (it && (p_line+1 < p_frame->lines.size()) && p_frame->lines[p_line+1].from==it) {

			if (p_mode==PROCESS_POINTER && r_click_item && p_click_pos.y>=p_ofs.y+y && p_click_pos.y<=p_ofs.y+y+lh) {
				//went to next line, but pointer was on the previous one
				if (r_outside) *r_outside=true;
				*r_click_item=itp;
				*r_click_char=rchar;
				return;
			}

			break;
		}
	}

	NEW_LINE;

#undef NEW_LINE
#undef ENSURE_WIDTH
#undef ADVANCE
#undef CHECK_HEIGHT

}
