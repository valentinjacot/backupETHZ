package jdraw.decorators;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Rectangle;

import jdraw.framework.Figure;

@SuppressWarnings("serial")
public class BorderedDecorator extends AbstractDecorator{

	public BorderedDecorator(Figure f) {
		super(f);
	}

	@Override
	public void draw(Graphics g) {
		Rectangle bounds = inner.getBounds();
		
		int x = bounds.x - 5;
		int y = bounds.y - 5;
		int w = bounds.width + 10 ;
		int h = bounds.height + 10;
		
		g.setColor(Color.white);
		g.drawLine(x, y, x + w, y);
		g.drawLine(x, y, x, y + h);
		g.setColor(Color.gray);
		g.drawLine(x + w, y, x + w , y + h);
		g.drawLine(x, y + h, x + w , y + h);
		inner.draw(g);

	}
	
	@Override 
	public Rectangle getBounds() {
		Rectangle bounds =inner.getBounds();
		bounds.grow(5,5);
		return bounds;
	}

	@Override
	public boolean contains(int x, int y) {
		return getBounds().contains(x,y);
	}

}
