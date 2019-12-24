package jdraw.decorators;

import java.awt.Graphics;
import java.awt.Rectangle;

import jdraw.framework.Figure;

@SuppressWarnings("serial")
public class AnimationDecorator extends AbstractDecorator{
	private boolean stop = true;
	public AnimationDecorator(Figure f) {
		super(f);
	}
	
	@Override
	public void draw(Graphics g) {
		Rectangle pan = inner.getBounds();
		int x = 1; int y = 1;
		int maxx = (int) pan.getX(), maxy = (int) pan.getY();
		boolean backX = false;
		boolean backY = false;
//		System.out.println(x + "   " + y);
		if (x < 1) 
			backX=false;
		if(x > 55) 
			backX = true;
		if (y < 1) 
			backY=false;
		if (y > 55) 
			backY=true;

		if(!backX) 
			x=1;
		else 
			x=-1;
		if(!backY) 
			y=1;
		else 
			y=-1;
		
		inner.move(x,y);
		
		while(!stop) {

			try {
				Thread.sleep(30000);
			}catch(InterruptedException e) {
				e.printStackTrace();
				stop = true;
			}
		}
		
	}

}
