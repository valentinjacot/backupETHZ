package logo.commands.turtle;

import java.util.LinkedList;

import logo.commands.Command;

public class CompositeCommand implements TurtleCommand {
	private final String name;
	private final LinkedList<TurtleCommand> commands = new LinkedList<>();

	public CompositeCommand(String name) {
		this.name = name;
	}

	public String getName() {
		return name;
	}

	public void add(TurtleCommand command) {
		commands.add(command);
	}

	@Override
	public void execute() {
		for (Command command : commands)
			command.execute();
	}

	@Override
	public String toString() {
		return "Macro " + name + ".";
	}

}
