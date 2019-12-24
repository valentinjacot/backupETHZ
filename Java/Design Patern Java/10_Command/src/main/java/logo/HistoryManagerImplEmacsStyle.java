package logo;

import java.util.LinkedList;

import logo.commands.Command;
import logo.commands.turtle.TurtleCommand;

public class HistoryManagerImplEmacsStyle implements HistoryManager {
	private final LogoInterpreter interpreter;
	private final LinkedList<Object> commands = new LinkedList<>();
	private int undoPosition = 0;
	
	public HistoryManagerImplEmacsStyle(LogoInterpreter interpreter) {
		this.interpreter = interpreter;
	}

	@Override
	public void addCommand(TurtleCommand command) {
		commands.add(command);
		undoPosition = commands.size();
	}

	@Override
	public void clear() {
		commands.clear();
		undoPosition = 0;
	}

	@Override
	public void undo() {
		if (undoPosition == 0) {
			System.out.println("Nothing left to undo");
			return;
		}
		undoPosition--;
		Undo u = new Undo(commands.get(undoPosition));
		commands.add(u);
		System.out.println((u.level % 2 == 0 ? "Redo " : "Undo ") + u.c);
		
		interpreter.repaint();
	}

	@Override
	public void redo() {
		if(commands.getLast() instanceof Command) {
			System.out.println("Nothing left to redo");
			return;
		}
		Undo u = (Undo)commands.removeLast();
		System.out.println((u.level % 2 == 0 ? "Undo " : "Redo ") + u.c);
		undoPosition++;
		
		interpreter.repaint();
	}
	
	@Override
	public Iterable<TurtleCommand> getCommands() {
		LinkedList<TurtleCommand> result = new LinkedList<>();
		for(Object c : commands) {
			if(c instanceof Undo) {
				Undo u = (Undo) c;
				if(u.level % 2 == 0) {
					result.add(u.c);
				} else {
					result.removeLast();
				}
			} else {
				result.add((TurtleCommand)c);
			}
		}
		return result;
	}
	
	static class Undo {
		private final TurtleCommand c;
		private final int level;

		public Undo(Object c) {
			if(c instanceof TurtleCommand) {
				this.c = (TurtleCommand)c;
				level = 1;
			} else {
				this.c = ((Undo)c).c;
				level = ((Undo)c).level+1;
			}
		}
		
		@Override
		public String toString() {
			return "Undo ["+ level + "] of " + c;
		}
	}

}
