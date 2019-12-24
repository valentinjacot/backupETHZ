package jdraw.commands;

import java.awt.Point;

import jdraw.framework.DrawCommand;
import jdraw.framework.Figure;

@SuppressWarnings("serial")
public class SetBoundsCommand implements DrawCommand{
	private final Figure figure;
	private Point fromOrigin, fromGoal, toOrigin, toGoal;
	
		
	public SetBoundsCommand(Figure figure, Point fromOrigin, Point fromGoal, Point toOrigin,
			Point toGoal) {
		this.figure = figure;
		this.fromOrigin = fromOrigin;
		this.fromGoal = fromGoal;
		this.toOrigin = toOrigin;
		this.toGoal = toGoal;
	}

	@Override
	public void redo() {
		figure.setBounds(fromOrigin,fromGoal);
	}

	@Override
	public void undo() {
		figure.setBounds(toOrigin,toGoal);
	}

}
