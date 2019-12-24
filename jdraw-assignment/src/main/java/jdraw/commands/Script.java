package jdraw.commands;

import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

import jdraw.framework.DrawCommand;

@SuppressWarnings("serial")
public class Script implements DrawCommand {
	List<DrawCommand> commands = new LinkedList<>();
	//CompositeDrawCommand ref sol
	
	
	@Override
	public void redo() {
		for ( DrawCommand dc: commands)
			dc.redo();
	}

	@Override
	public void undo() {
//		for ( DrawCommand dc: commands)
//			dc.undo();
		int size = commands.size();
		ListIterator<DrawCommand> it = commands.listIterator(size);
		while (it.hasPrevious()) { it.previous().undo(); }
	}

	public void addCommand(DrawCommand cmd) {
		commands.add(cmd);
	}

}
