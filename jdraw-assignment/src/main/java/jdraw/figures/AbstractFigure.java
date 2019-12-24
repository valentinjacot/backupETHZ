package jdraw.figures;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import jdraw.framework.Figure;
import jdraw.framework.FigureEvent;
import jdraw.framework.FigureListener;


@SuppressWarnings("serial")
public abstract class AbstractFigure implements Figure{
	private List<FigureListener> FListeners = new CopyOnWriteArrayList<>();
	
	
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
	
	/**
	 * Returns a list of 8 handles for this Rectangle.
	 * @return all handles that are attached to the targeted figure.
	 * @see jdraw.framework.Figure#getHandles()
	 */	

		

	@Override
	public Figure clone() {
		try {
			AbstractFigure copy = (AbstractFigure) super.clone();
			copy.FListeners = new CopyOnWriteArrayList<>();
			return copy;
		}catch (CloneNotSupportedException e ) {
			System.out.println("clone not supported exception");
			throw new InternalError();
		}
	}
	
	public abstract void swapHorizontal() ;
	
	public abstract void swapVertical() ;


}
