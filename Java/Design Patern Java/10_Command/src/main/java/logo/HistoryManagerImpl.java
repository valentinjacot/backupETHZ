package logo;

import java.util.Stack;

import logo.HistoryManager;
import logo.LogoInterpreter;
import logo.commands.turtle.TurtleCommand;

public class HistoryManagerImpl implements HistoryManager {
	private final LogoInterpreter interpreter;
	private final Stack<TurtleCommand> undoStack = new Stack<>();
	private final Stack<TurtleCommand> redoStack = new Stack<>();
	
	public HistoryManagerImpl(LogoInterpreter interpreter) {
		this.interpreter = interpreter;
	}

	@Override
	public void addCommand(TurtleCommand command) {
		undoStack.push(command);
		redoStack.clear();
	}

	@Override
	public void clear() {
		undoStack.clear();
		redoStack.clear();
	}

	@Override
	public void undo() {
		if (undoStack.isEmpty()) {
			System.out.println("Nothing left to undo");
			return;
		}
		TurtleCommand command = undoStack.pop();
		System.out.println("Undo " + command);
		redoStack.push(command);

		interpreter.repaint();
	}

	@Override
	public void redo() {
		if (redoStack.isEmpty()) {
			System.out.println("Nothing left to redo");
			return;
		}
		TurtleCommand command = redoStack.pop();
		System.out.println("" + command);
		command.execute();
		undoStack.push(command);
	}
	
	@Override
	public Iterable<TurtleCommand> getCommands() { 
		return undoStack;
	}

}
