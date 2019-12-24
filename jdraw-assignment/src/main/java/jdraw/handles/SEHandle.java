package jdraw.handles;

import java.awt.Cursor;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.event.MouseEvent;

import jdraw.figures.AbstractFigure;
import jdraw.framework.DrawView;
import jdraw.framework.Figure;

public class SEHandle extends AbstractHandleState{

	public SEHandle(Figure f_) {
		super(f_);
	}
	@Override
	public Point getLocation() {
		Rectangle r = this.owner.getBounds();
		return new Point(r.x+r.width, r.y+r.height);
	}

	@Override
	public Cursor getCursor() {
		return Cursor.getPredefinedCursor(Cursor.SE_RESIZE_CURSOR);
	}

	@Override
	public void dragIteraction(int x, int y, MouseEvent e, DrawView v) {
		Rectangle r = this.getOwner().getBounds();
		this.getOwner().setBounds(new Point(r.x, r.y), new Point(x,y));
		if (x < r.x) {
			((AbstractFigure) owner).swapHorizontal();
		} else if ( y < r.y) {
			((AbstractFigure) owner).swapVertical();
		}
	}

}
