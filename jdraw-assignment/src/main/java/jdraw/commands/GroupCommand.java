package jdraw.commands;

import jdraw.figures.Group;
import jdraw.framework.DrawCommand;
import jdraw.framework.DrawModel;
import jdraw.framework.Figure;

@SuppressWarnings("serial")
public class GroupCommand implements DrawCommand{
	private final Group group;
	private final DrawModel model;
	private final boolean insertGroup;
	
	public GroupCommand(Group group, DrawModel model, boolean insertGroup) {
		super();
		this.group = group;
		this.model = model;
		this.insertGroup = insertGroup;
	}

	@Override
	public void redo() {
		if(insertGroup) {insertGroup();}else {removeGroup();}
	}

	@Override
	public void undo() {
		if(insertGroup) {removeGroup();}else {insertGroup();}
	}
	
	public void insertGroup() {
		for (Figure f: group.getFigureParts()) {
			model.removeFigure(f);
		}
		model.addFigure((Figure)group);
	};
	public void removeGroup() {
		model.removeFigure((Figure)group);
		for (Figure f: group.getFigureParts()) {
			model.addFigure(f);
		}
	};

}
