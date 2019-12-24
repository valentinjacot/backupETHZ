/*
 * Copyright (c) 2018 Fachhochschule Nordwestschweiz (FHNW)
 * All Rights Reserved. 
 */

package jdraw.std;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import jdraw.commands.ListDrawCommandHandler;
import jdraw.framework.DrawCommandHandler;
import jdraw.framework.DrawModel;
import jdraw.framework.DrawModelEvent;
import jdraw.framework.DrawModelListener;
import jdraw.framework.Figure;
import jdraw.framework.FigureEvent;
import jdraw.framework.FigureListener;
//import jdraw.framework.DrawModelEvent.Type;
//import jdraw.framework.DrawModelEvent;
/**
 * Provide a standard behavior for the drawing model. This class initially does not implement the methods
 * in a proper way.
 * It is part of the course assignments to do so.
 * @author Valentin Jacot-Descombes
 *
 */
public class StdDrawModel implements DrawModel, FigureListener {
	
	private List<DrawModelListener> DMListeners = new CopyOnWriteArrayList<>();
	private List<Figure> figures = new LinkedList<>();
	
	@Override
	public void addFigure(Figure f) {
		if(!figures.contains(f) && !f.equals(null)) {
			figures.add(f);
			notifyDMListeners(f,DrawModelEvent.Type.FIGURE_ADDED);
			f.addFigureListener(this);
		}
	}

	@Override
	public Iterable<Figure> getFigures() {
		return Collections.unmodifiableList(figures);
	}

	@Override
	public void removeFigure(Figure f) {
		if(figures.remove(f) && !f.equals(null)) {
			notifyDMListeners(f,DrawModelEvent.Type.FIGURE_REMOVED);
			f.removeFigureListener(this);
		}
	}

	@Override
	public void addModelChangeListener(DrawModelListener listener) {
		if (!DMListeners.contains(listener) && !listener.equals(null))
			DMListeners.add(listener);
	}

	@Override
	public void removeModelChangeListener(DrawModelListener listener) {
		if (!listener.equals(null))
			DMListeners.remove(listener);
	}
	
	public void notifyDMListeners(Figure f, DrawModelEvent.Type type) {
		DrawModelEvent dme = new DrawModelEvent(this, f, type);
		for(DrawModelListener l : DMListeners) {
			l.modelChanged(dme);
		}
	}

	@Override
	public void setFigureIndex(Figure f, int index) {
		if(!figures.contains(f))
			throw(new IllegalArgumentException());
		else if(index < 0 || index >= figures.size())
			throw(new IndexOutOfBoundsException());
		else {
			int originalIndex = figures.indexOf(f);
			figures.remove(f);
			figures.add(index,f);
			if (index != originalIndex)
				notifyDMListeners(null, DrawModelEvent.Type.DRAWING_CHANGED);
			
		}
		
	}

	@Override
	public void removeAllFigures() {
		for (Figure f:figures) {
			f.removeFigureListener(this);
		}
		figures = new LinkedList<Figure>();
		//figures.removeAll();
		notifyDMListeners(null, DrawModelEvent.Type.DRAWING_CLEARED);
	}

	@Override
	public void figureChanged(FigureEvent e) {
		notifyDMListeners(e.getFigure(),  DrawModelEvent.Type.FIGURE_CHANGED);
	}
	
	
	
	/** The draw command handler. Initialized here with a dummy implementation. */
	// TODO initialize with your implementation of the undo/redo-assignment.
	private DrawCommandHandler handler = new ListDrawCommandHandler();

	/**
	 * Retrieve the draw command handler in use.
	 * @return the draw command handler.
	 */
	@Override
	public DrawCommandHandler getDrawCommandHandler() {
		return handler;
	}

}
