package jdraw.handles;

import java.awt.Cursor;
import java.awt.Point;
import java.awt.event.MouseEvent;

import jdraw.framework.DrawView;
import jdraw.framework.Figure;

public abstract class AbstractHandleState implements HandleState {
	protected Figure owner;
	
	public AbstractHandleState(Figure f_) {
		this.owner = f_;
	}


	@Override
	public Figure getOwner() {
		return this.owner;
	}

	@Override
	public abstract Point getLocation();	

	@Override
	public abstract Cursor getCursor();

	@Override
	public abstract void dragIteraction(int x, int y, MouseEvent e, DrawView v);

}
