package logo.commands.turtle;

import logo.LogoInterpreter;
import logo.commands.control.ControlCommand;

public class ClearScreenCommand implements ControlCommand {
	private final LogoInterpreter interpreter;

	public ClearScreenCommand(LogoInterpreter interpreter) {
		this.interpreter = interpreter;
	}

	@Override
	public void execute() {
		interpreter.resetTurtle();
		interpreter.getHistoryManager().clear();
	}

	@Override
	public String toString() {
		return "Resetting graphics.";
	}

}
