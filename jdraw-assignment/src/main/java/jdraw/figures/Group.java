	package jdraw.figures;

import java.awt.Graphics;
import java.awt.Point;
import java.awt.Rectangle;
import java.util.LinkedList;
import java.util.List;

import jdraw.framework.Figure;
import jdraw.framework.FigureGroup;
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
public class Group extends AbstractFigure implements FigureGroup {
	List<Figure> parts = new LinkedList<>();
	List<FigureHandle> FHandles = new LinkedList<>();
	
	public Group(List<Figure> inputFigures) {
		parts = new LinkedList<>(inputFigures);
		setHandles();
	}

	@Override
	public Iterable<Figure> getFigureParts() {
		return parts;
	}

	@Override
	public void draw(Graphics g) {
		for (Figure p:parts)
			p.draw(g);
	}

	@Override
	public void move(int dx, int dy) {
		if(dy!=0||dx!=0) {
			for (Figure p:parts)
				p.move(dx,dy);
		}
		notifyFListeners();
	}

	@Override
	public boolean contains(int x, int y) {
		Rectangle temp = this.getBounds();
		return temp.contains(x,y);
	}

	@Override
	public void setBounds(Point origin, Point corner) {
		System.out.println("set bounds not implemented");
//		for (Figure p:parts)
//			p.setBounds(origin,corner);
	}

	@Override
	public Rectangle getBounds() {
		Rectangle bounds= null;
		for (Figure p:parts)
			if (bounds == null)
				bounds = p.getBounds();
			else
				bounds.add(p.getBounds());
		return bounds;
	}
	
	@Override
	public Group clone() {
		Group copy = (Group) super.clone();
		copy.parts = new LinkedList<Figure>();

		for (Figure f: parts) {
			copy.parts.add(f);
		}
		return copy;
	}

	@Override
	public List<FigureHandle> getHandles() {
		return FHandles;
	}

	
	private void setHandles() {
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
