package jdraw.handles;

import java.awt.Cursor;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.event.MouseEvent;

import jdraw.figures.AbstractFigure;
import jdraw.framework.DrawView;
import jdraw.framework.Figure;

public class NHandle extends AbstractHandleState {

	public NHandle(Figure f_) {
		super(f_);
	}

	@Override
	public Point getLocation() {
		Rectangle r = this.owner.getBounds();
		return new Point(r.x + r.width/2, r.y);
	}

	@Override
	public Cursor getCursor() {
		return Cursor.getPredefinedCursor(Cursor.N_RESIZE_CURSOR);
	}

	@Override
	public void dragIteraction(int x, int y, MouseEvent e, DrawView v) {
		Rectangle r = this.getOwner().getBounds();
		this.getOwner().setBounds(new Point(r.x, y), new Point(r.x + r.width,r.y +r.height ));
		if (y > r.y + r.height) {
			((AbstractFigure) owner).swapVertical();
		}
	}

}
