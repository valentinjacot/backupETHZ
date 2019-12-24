package logo;

import java.util.ArrayList;
import java.util.Collections;

import logo.commands.turtle.TurtleCommand;

public class HistoryManagerImplList implements HistoryManager {
	private final LogoInterpreter interpreter;
	
	public HistoryManagerImplList(LogoInterpreter interpreter) {
		this.interpreter = interpreter;
	}
	
	private ArrayList<TurtleCommand> commands = new ArrayList<>();
	private int nOfUndoableCommands = 0;
	
	@Override
	public void addCommand(TurtleCommand command) {
		if(nOfUndoableCommands < commands.size()) {
			commands = new ArrayList<>(commands.subList(0, nOfUndoableCommands));
		}
		commands.add(command);
		nOfUndoableCommands++;
	}

	@Override
	public void clear() {
		commands.clear();
		nOfUndoableCommands = 0;
	}

	@Override
	public void undo() {
		if (nOfUndoableCommands == 0) {
			System.out.println("undo not possible");
		} else {
			nOfUndoableCommands--;
			interpreter.repaint();
		}
	}

	@Override
	public void redo() {
		if (nOfUndoableCommands == commands.size()) {
			System.out.println("redo not possible");
		} else {
			commands.get(nOfUndoableCommands).execute();
			nOfUndoableCommands++;
		}
	}

	@Override
	public Iterable<TurtleCommand> getCommands() {
		return Collections.unmodifiableList(commands.subList(0, nOfUndoableCommands));
	}

}
