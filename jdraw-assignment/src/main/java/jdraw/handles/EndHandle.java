package jdraw.handles;

import java.awt.Cursor;
import java.awt.Point;
import java.awt.event.MouseEvent;

import jdraw.figures.Line;
import jdraw.framework.DrawView;
import jdraw.framework.Figure;

public class EndHandle extends AbstractHandleState {

	public EndHandle(Figure f_) {
		super(f_);
	}

	@Override
	public Point getLocation() {
		Point p =((Line)owner).getP2();
		return new Point((int)p.getX(),(int)p.getY());
	}

	@Override
	public Cursor getCursor() {
		return Cursor.getPredefinedCursor(Cursor.MOVE_CURSOR);
	}

	@Override
	public void dragIteraction(int x, int y, MouseEvent e, DrawView v) {
		Point p =((Line)owner).getP1();
		this.getOwner().setBounds(new Point(x, y), p);
	}

}	