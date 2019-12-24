package jdraw.decorators;

import java.awt.Cursor;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.event.MouseEvent;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import jdraw.framework.DrawView;
import jdraw.framework.Figure;
import jdraw.framework.FigureEvent;
import jdraw.framework.FigureHandle;
import jdraw.framework.FigureListener;

@SuppressWarnings("serial")
public abstract class AbstractDecorator implements Figure, FigureListener{
	protected Figure inner;
	private List<FigureListener> FListeners = new CopyOnWriteArrayList<>();

	
	public AbstractDecorator(Figure f) {
		inner = f;
		construct();
	}
	
	public void construct() {
		inner.addFigureListener(this);
	}
	
	public void destruct() {
		inner.removeFigureListener(this);
	}
	@Override
	public void draw(Graphics g) {
		inner.draw(g);
	}

	@Override
	public void move(int dx, int dy) {
		inner.move(dx,dy);
	}

	@Override
	public boolean contains(int x, int y) {
		return inner.contains(x,y);
	}

	@Override
	public void setBounds(Point origin, Point corner) {
		inner.setBounds(origin, corner);
	}

	@Override
	public Rectangle getBounds() {
		return inner.getBounds();
	}

	@Override
	public List<FigureHandle> getHandles() {
		return inner.getHandles();
	}

	@Override
	public void addFigureListener(FigureListener listener) {
		if (!FListeners.contains(listener) && !listener.equals(null))
			FListeners.add(listener);
	}

	@Override
	public void removeFigureListener(FigureListener listener) {
		if (!listener.equals(null))
			FListeners.remove(listener);
	}
	
	public void notifyFListeners() {
		FigureEvent e =new FigureEvent(this);
		for (FigureListener fl:FListeners)
			fl.figureChanged(e);
	}
	
	@Override
	public AbstractDecorator clone() {
		try {
			AbstractDecorator f = (AbstractDecorator) super.clone();
			f.FListeners  = new CopyOnWriteArrayList<>();
			f.inner = (Figure) inner.clone();
			return f;
		} catch (CloneNotSupportedException e) {
            throw new InternalError();
        }
    }

	@Override
	public void figureChanged(FigureEvent e) {
		for (FigureListener fl:FListeners)
			fl.figureChanged(new FigureEvent(this));
	}
//	public abstract void swapHorizontal() ;
//	
//	public abstract void swapVertical() ;

	public final class DecoratorHandle implements FigureHandle{
		private FigureHandle fh;
		public DecoratorHandle(FigureHandle fh) {
			this.fh = fh;
		}
		@Override
		public Figure getOwner() {
			return AbstractDecorator.this;
		}
		@Override
		public Point getLocation() {
			return fh.getLocation();
		}
		@Override
		public void draw(Graphics g) {
			fh.draw(g);
		}
		@Override
		public Cursor getCursor() {
			return fh.getCursor();
		}
		@Override
		public boolean contains(int x, int y) {
			return fh.contains(x, y);
		}
		@Override
		public void startInteraction(int x, int y, MouseEvent e, DrawView v) {
			fh.startInteraction(x,y, e,v);
		}
		@Override
		public void dragInteraction(int x, int y, MouseEvent e, DrawView v) {
			fh.dragInteraction(x,y,e,v);
		}
		@Override
		public void stopInteraction(int x, int y, MouseEvent e, DrawView v) {
			fh.stopInteraction(x,y,e,v);
		}
		
	}
}
