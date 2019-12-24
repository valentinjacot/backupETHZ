package logo.commands.turtle;

public class RepeatCommand implements TurtleCommand {
	private final int amount;
	private final TurtleCommand command;

	public RepeatCommand(int amount, TurtleCommand command) {
		this.amount = amount;
		this.command = command;
	}

	@Override
	public void execute() {
		for (int i = 0; i < amount; i++) {
			command.execute();
		}
	}

	@Override
	public String toString() {
		return "Repeating " + amount + " times " + command;
	}

}
