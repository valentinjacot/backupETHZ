package jdraw.commands;

import jdraw.framework.DrawCommand;
import jdraw.framework.DrawModel;
import jdraw.framework.Figure;

@SuppressWarnings("serial")
public class AddFigureCommand implements DrawCommand{
	private final DrawModel model;
	private final Figure figure;
//	private int index;
	
	public AddFigureCommand(DrawModel dm, Figure f) {
		this.model = dm;
		this.figure = f;
//		index = dm.getFigures().indexOf(f);
//		if (index == -1) throw new IllegalArgumentException();
	}
	
	
	@Override
	public void redo() {
		model.addFigure(figure);
//		model.setFigureIndex(figure, index);
	}

	@Override
	public void undo() {
		model.removeFigure(figure);

	}

}
