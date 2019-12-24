package logo.commands.control;

public class ErrorCommand implements ControlCommand {
	private final String msg;

	public ErrorCommand(String msg) {
		this.msg = msg;
	}

	@Override
	public void execute() {
		// do nothing
	}

	@Override
	public String toString() {
		return "Error: " + msg;
	}
}