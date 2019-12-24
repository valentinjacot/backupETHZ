package logo.commands.control;

public class NotFoundCommand implements ControlCommand {
	private final String commandName;

	public NotFoundCommand(String commandName) {
		this.commandName = commandName;
	}

	@Override
	public void execute() {
		// do nothing
	}

	@Override
	public String toString() {
		return "Command " + commandName + " unknown.";
	}
}