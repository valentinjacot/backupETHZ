package jdraw.handles;

import java.awt.Cursor;
import java.awt.Point;
import java.awt.event.MouseEvent;

import jdraw.figures.Line;
import jdraw.framework.DrawView;
import jdraw.framework.Figure;

public class BeginHandle extends AbstractHandleState {

	public BeginHandle(Figure f_) {
		super(f_);
	}

	@Override
	public Point getLocation() {
		Point p =((Line)owner).getP1();
		return p;
	}

	@Override
	public Cursor getCursor() {
		return Cursor.getPredefinedCursor(Cursor.MOVE_CURSOR);
	}

	@Override
	public void dragIteraction(int x, int y, MouseEvent e, DrawView v) {
		Point p = ((Line)owner).getP1();
		this.getOwner().setBounds(new Point(x, y), p);
	}

}
