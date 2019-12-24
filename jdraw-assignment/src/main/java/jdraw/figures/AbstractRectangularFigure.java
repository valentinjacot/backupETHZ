package jdraw.figures;

import java.awt.Point;
import java.awt.Rectangle;
import java.util.LinkedList;
import java.util.List;

import jdraw.framework.Figure;
import jdraw.framework.FigureHandle;
import jdraw.handles.EHandle;
import jdraw.handles.Handle;
import jdraw.handles.HandleState;
import jdraw.handles.NEHandle;
import jdraw.handles.NHandle;
import jdraw.handles.NWHandle;
import jdraw.handles.SEHandle;
import jdraw.handles.SHandle;
import jdraw.handles.SWHandle;
import jdraw.handles.WHandle;

@SuppressWarnings("serial")
public abstract class AbstractRectangularFigure extends AbstractFigure{
	protected Rectangle rectangle;
	private List<FigureHandle> FHandles= new LinkedList<>();

	
	public AbstractRectangularFigure(int x, int y, int w, int h) {
		rectangle = new Rectangle(x, y, w, h);
		setHandles();
	}

	@Override
	public void setBounds(Point origin, Point corner) {
		if( origin != rectangle.getLocation()) {
			rectangle.setFrameFromDiagonal(origin, corner);
			notifyFListeners();
		}
	}

	@Override
	public void move(int dx, int dy) {
		if( dx!=0 || dy!=0) {
			rectangle.setLocation(rectangle.x + dx, rectangle.y + dy);
			notifyFListeners();
		}
	}

	@Override
	public Rectangle getBounds() {
		return rectangle.getBounds();
	}
	
	@Override
	public 	List<FigureHandle> getHandles() {		
		return FHandles;
	}
	@Override
	public boolean contains(int x, int y) {
		return rectangle.contains(x, y);
	}
	
	@Override
	public Figure clone() {
		try {
			AbstractRectangularFigure copy = (AbstractRectangularFigure) super.clone();
			copy.rectangle = (Rectangle) this.rectangle.clone();
			copy.setHandles();
			return copy;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();		return null;

		}
	}
	
	protected void setHandles() {
		FHandles.add(new Handle(new NWHandle(this)));
		FHandles.add(new Handle(new NEHandle(this)));
		FHandles.add(new Handle(new SWHandle(this)));
		FHandles.add(new Handle(new SEHandle(this)));
		FHandles.add(new Handle(new NHandle(this)));
		FHandles.add(new Handle(new SHandle(this)));
		FHandles.add(new Handle(new EHandle(this)));
		FHandles.add(new Handle(new WHandle(this)));
	}
	
	@Override 
	public void swapHorizontal() {
		Handle NW = (Handle) FHandles.get(0);
		Handle NE = (Handle) FHandles.get(1);
		Handle SW = (Handle) FHandles.get(2);
		Handle SE = (Handle) FHandles.get(3);
		Handle E = (Handle) FHandles.get(6);
		Handle W = (Handle) FHandles.get(7);
		HandleState NWState = NW.getState();
		HandleState NEState = NE.getState();
		HandleState SWState = SW.getState();
		HandleState SEState = SE.getState();
		HandleState EState = E.getState();
		HandleState WState = W.getState();
		NW.setState(NEState);
		NE.setState(NWState);
		SE.setState(SWState);
		SW.setState(SEState);
		E.setState(WState);
		W.setState(EState);
		
	}
	

	@Override
	public void swapVertical() {	
		Handle NW = (Handle) FHandles.get(0);
		Handle NE = (Handle) FHandles.get(1);
		Handle SW = (Handle) FHandles.get(2);
		Handle SE = (Handle) FHandles.get(3);
		Handle N = (Handle) FHandles.get(4);
		Handle S = (Handle) FHandles.get(5);
		HandleState NWState = NW.getState();
		HandleState NEState = NE.getState();
		HandleState SWState = SW.getState();
		HandleState SEState = SE.getState();
		HandleState NState = N.getState();
		HandleState SState = S.getState();
		NW.setState(SWState);
		NE.setState(SEState);
		SE.setState(NEState);
		SW.setState(NWState);
		N.setState(SState);
		S.setState(NState);		
	}


}
