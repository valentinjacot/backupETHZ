package logo.commands.control;

import logo.HistoryManager;

public class UndoCommand extends AbstractHistoryCommand {

	public UndoCommand(HistoryManager historyManager) {
		super(historyManager);
	}

	@Override
	public void execute() {
		getHistoryManager().undo();
	}

	@Override
	public String toString() {
		return "Undoing command...";
	}

}
