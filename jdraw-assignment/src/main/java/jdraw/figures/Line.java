package jdraw.figures;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.geom.Line2D;
import java.util.LinkedList;
import java.util.List;

import jdraw.framework.FigureHandle;
import jdraw.handles.BeginHandle;
import jdraw.handles.EndHandle;
import jdraw.handles.Handle;



@SuppressWarnings("serial")
public class Line extends AbstractFigure{
	private Line2D line;
	private List<FigureHandle> FHandles= new LinkedList<>();
	
	public Line(double x0, double y0, double x1, double y1) {
		line  = new Line2D.Double(x0,y0,x1,y1);
	}
	
	@Override
	public void draw(Graphics g) {
		g.setColor(Color.BLACK);
		g.drawLine((int)line.getX1(),(int)line.getY1(),(int)line.getX2(),(int)line.getY2());
	}

	@Override
	public void move(int dx, int dy) {
		// TODO Auto-generated method stub
		line.setLine(line.getX1()+dx,line.getY1()+dy,line.getX2()+dx, line.getY2()+dy);
		notifyFListeners();
	}

	@Override
	public void setBounds(Point origin, Point corner) {
		if( origin != line.getP1()|| origin != line.getP2()) {
			line.setLine(origin.getX(),origin.getY(),corner.getX(),corner.getY());
			notifyFListeners();
		}		
	}
	@Override
	public Rectangle getBounds() {
		return line.getBounds();
	}
	
	public Point getP1() {
		return (Point) line.getP1();
	}	
	public Point getP2() {
		return (Point) line.getP2();
	}
	
	@Override
	public Line clone() {
		Line copy = (Line) super.clone();
		copy.setHandles();
		copy.line = (Line2D) this.line.clone();
		return copy;
	}
	@Override
	public boolean contains(int x, int y) {
		return line.ptSegDist(x,y) <= 5;
	}

	@Override
	public List<FigureHandle> getHandles() {
		return FHandles;
	}
	
	protected void setHandles() {
		FHandles.add(new Handle(new BeginHandle(this)));		
		FHandles.add(new Handle(new EndHandle(this)));		
	}
	@Override
	public void swapHorizontal() {}

	@Override
	public void swapVertical() {}

}
