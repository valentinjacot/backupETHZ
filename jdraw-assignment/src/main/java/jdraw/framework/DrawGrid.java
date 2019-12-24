/*
 * Copyright (c) 2018 Fachhochschule Nordwestschweiz (FHNW)
 * All Rights Reserved. 
 */

package jdraw.framework;

import java.awt.Point;

/**
 * A DrawGrid is used to restrict the coordinates used when the mouse is
 * clicked. The methods defined in this interface are used by the draw view.
 * 
 * @author Dominik Gruntz &amp; Christoph Denzler
 * @version 2.5
 */
public interface DrawGrid {
	
	/**
	 * Returns constrained coordinates for p, e.g. rounded to a grid.
	 * 
	 * @param p mouse coordinates
	 * @return constrained coordinates
	 */
	Point constrainPoint(Point p);

	/**
	 * Returns the horizontal step size when the selection is moved with the arrow
	 * keys.
	 * 
	 * @param right true if selection is moved right; false otherwise
	 * @return step size in horizontal direction (positive result)
	 */
	int getStepX(boolean right);

	/**
	 * Returns the vertical step size when the selection is moved with the arrow
	 * keys.
	 * 
	 * @param down true if selection is moved down; false otherwise
	 * @return step size in vertical direction (positive result)
	 */
	int getStepY(boolean down);
	
	/**
	 * Activates the grid. This method is called whenever method setGrid is called
	 * on a draw view.
	 */
	void activate();

	/**
	 * Deactivates the grid. This method is called whenever another grid is
	 * installed. Use this method to do some clean-up when the grid is switched.
	 */
	void deactivate();

	/**
	 * Indicates that a mouse interaction was just started. This method might be
	 * used to setup data used during this mouse interaction.
	 */
	void mouseDown();

	/**
	 * Indicates that a mouse interaction has been finished. This method might be
	 * used to clean-up data created in method mouseDown.
	 */
	void mouseUp();

}
