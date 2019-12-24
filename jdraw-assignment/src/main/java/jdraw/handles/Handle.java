package jdraw.handles;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.event.MouseEvent;

import jdraw.commands.SetBoundsCommand;
import jdraw.framework.DrawView;
import jdraw.framework.Figure;
import jdraw.framework.FigureHandle;

public class Handle implements FigureHandle{
//	private Figure owner;
	private final int SIZE = 6;
	
	public HandleState state;

    private Point originFrom = null;
    private Point cornerFrom = null;
    
	Rectangle oldBounds;
	Rectangle newBounds;
	
	public Handle(HandleState hs) {
		this.state = hs;
	}

	@Override
	public Figure getOwner() {
		return state.getOwner();
	}

	@Override
	public Point getLocation() {
		return state.getLocation();
	}

	@Override
	public void draw(Graphics g) {
		Point loc = getLocation();
		g.setColor(Color.WHITE);g.fillRect(loc.x - SIZE/2,loc.y- SIZE/2, SIZE, SIZE);
		g.setColor(Color.BLACK);g.drawRect(loc.x - SIZE/2,loc.y- SIZE/2, SIZE, SIZE);
	}

	@Override
	public Cursor getCursor() {
		return state.getCursor();
	}

	@Override
	public boolean contains(int x, int y) {
		Point loc = getLocation();
		return (x > loc.x - SIZE/2 && x < loc.x + SIZE/2 
				&& y > loc.y - SIZE/2 && y < loc.y + SIZE/2);
	}

	@Override
	public void startInteraction(int x, int y, MouseEvent e, DrawView v) {
		oldBounds = new Rectangle(getOwner().getBounds());		
	    originFrom = new Point((int) oldBounds.getX(), (int) oldBounds.getY());
	    cornerFrom = new Point((int) oldBounds.getX() + (int) oldBounds.getWidth(), (int) oldBounds.getY() + (int) oldBounds.getHeight());
	}

	@Override
	public void dragInteraction(int x, int y, MouseEvent e, DrawView v) {
		state.dragIteraction( x,  y,  e,  v);		
	}

	@Override
	public void stopInteraction(int x, int y, MouseEvent e, DrawView v) {
		newBounds = new Rectangle(getOwner().getBounds());
        Point originTo = new Point((int) newBounds.getX(), (int) newBounds.getY());
        Point cornerTo = new Point((int) newBounds.getX() + (int) newBounds.getWidth(), (int) newBounds.getY() + (int) newBounds.getHeight());

		v.getModel().getDrawCommandHandler().addCommand(
				new SetBoundsCommand(getOwner(), originTo, cornerTo, originFrom, cornerFrom)
				);
	}
	
	public HandleState getState() {
		return state;
	}
	public void setState(HandleState hs) {
		this.state = hs;
	}

}
