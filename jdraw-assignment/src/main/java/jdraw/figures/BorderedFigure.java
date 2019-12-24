package jdraw.figures;

import java.awt.Graphics;
import java.awt.Point;
import java.awt.Rectangle;
import java.util.List;

import jdraw.framework.Figure;
import jdraw.framework.FigureHandle;
import jdraw.framework.FigureListener;

@SuppressWarnings("serial")
public class BorderedFigure extends AbstractFigure {
	private Figure inner;

	public BorderedFigure(Figure inner) {
		super();
		this.inner = inner;
	}

	public void draw(Graphics g) {
		inner.draw(g);
	}

	public void move(int dx, int dy) {
		inner.move(dx, dy);
	}

	public boolean contains(int x, int y) {
		return inner.contains(x, y);
	}

	public void setBounds(Point origin, Point corner) {
		inner.setBounds(origin, corner);
	}

	public Rectangle getBounds() {
		return inner.getBounds();
	}

	public List<FigureHandle> getHandles() {
		return inner.getHandles();
	}

	public void addFigureListener(FigureListener listener) {
		inner.addFigureListener(listener);
	}

	public void removeFigureListener(FigureListener listener) {
		inner.removeFigureListener(listener);
	}

	public Figure clone() {
		return inner.clone();
	}

	@Override
	public void swapHorizontal() {
		// TODO Auto-generated method stub
	}

	@Override
	public void swapVertical() {
		// TODO Auto-generated method stub
		
	}
	
}
