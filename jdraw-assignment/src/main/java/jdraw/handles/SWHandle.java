package jdraw.handles;

import java.awt.Cursor;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.event.MouseEvent;

import jdraw.figures.AbstractFigure;
import jdraw.framework.DrawView;
import jdraw.framework.Figure;

public class SWHandle extends AbstractHandleState{

	public SWHandle(Figure f_) {
		super(f_);
	}
	@Override
	public Point getLocation() {
		Rectangle r = this.owner.getBounds();
		return new Point(r.x, r.y+r.height);
	}

	@Override
	public Cursor getCursor() {
		return Cursor.getPredefinedCursor(Cursor.SW_RESIZE_CURSOR);
	}

	@Override
	public void dragIteraction(int x, int y, MouseEvent e, DrawView v) {
		Rectangle r = this.getOwner().getBounds();
		this.getOwner().setBounds(new Point(x, r.y), new Point(r.x+r.width,y));
		if (x > r.x + r.width) {
			((AbstractFigure) owner).swapHorizontal();
		} else if ( y < r.y) {
			((AbstractFigure) owner).swapVertical();
		}
	}

}
