package jdraw.constrainers;

import java.awt.Point;

import jdraw.framework.DrawGrid;

public class GridFix implements DrawGrid{
	private final int stepX;
	private final int stepY;
	
	public GridFix(int x, int y) {
		stepX=x;
		stepY=y;
	}
	@Override
	public Point constrainPoint(Point p) {
		double x = p.getX();
		double y = p.getY();
		double clx=((x % stepX < stepX/2)? x - x % stepX : x + (stepX - (x % stepX) )); //most difficult possible way to find the grid point
		double cly=((y % stepY < stepY/2)? y - y % stepY : y + (stepY - (y % stepY) ));
//		double clx = Math.round((float)x/stepX)*stepX;
//		double cly = Math.round((float)y/stepY)*stepY;
		return new Point((int)clx,(int)cly);
	}

	@Override
	public int getStepX(boolean right) {
		return stepX;
	}

	@Override
	public int getStepY(boolean down) {
		return stepY;
	}

	@Override
	public void activate() {
		System.out.println("GridFix:activate");
	}

	@Override
	public void deactivate() {
		System.out.println("GridFix:deactivate");		
	}

	@Override
	public void mouseDown() {
		System.out.println("GridFix:mouseDown");		
	}

	@Override
	public void mouseUp() {
		System.out.println("GridFix:mouseUp");		
	}
	

}
