package jdraw.handles;

import java.awt.Cursor;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.event.MouseEvent;

import jdraw.figures.AbstractFigure;
import jdraw.framework.DrawView;
import jdraw.framework.Figure;

public class NEHandle extends AbstractHandleState {

	public NEHandle(Figure f_) {
		super(f_);
	}
	@Override
	public Point getLocation() {
		Rectangle r = this.owner.getBounds();
		return new Point(r.x+r.width, r.y);
	}

	@Override
	public Cursor getCursor() {
		return Cursor.getPredefinedCursor(Cursor.NE_RESIZE_CURSOR);
	}

	@Override
	public void dragIteraction(int x, int y, MouseEvent e, DrawView v) {
		Rectangle r = this.getOwner().getBounds();
		this.getOwner().setBounds(new Point(r.x, y), new Point(x,r.y+r.height));
		if (x < r.x) {
			((AbstractFigure) owner).swapHorizontal();
		} else if ( y > r.y + r.height) {
			((AbstractFigure) owner).swapVertical();
		}
	}
}
