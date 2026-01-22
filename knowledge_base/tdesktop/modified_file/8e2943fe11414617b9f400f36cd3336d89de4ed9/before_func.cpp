
	case QEvent::TouchCancel:
		_touchPress = false;
		_touchTimer.stop();
		break;
	}
}

QRect FlatTextarea::getTextRect() const {
	return rect().marginsRemoved(_st.textMrg + st::textRectMargins);
}

int32 FlatTextarea::fakeMargin() const {
	return _fakeMargin;
}

void FlatTextarea::paintEvent(QPaintEvent *e) {
	QPainter p(viewport());
	p.fillRect(rect(), _st.bgColor->b);
