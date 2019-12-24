package logo.parser;

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import logo.commands.Command;
import logo.commands.CommandFactory;
import logo.commands.control.NotFoundCommand;

public class CommandRegistry {
	private Map<String, CommandFactory> commandFactoryByName = new HashMap<>();

	public void registerCommand(String name, CommandFactory factory) {
		commandFactoryByName.put(name, factory);
	}

	public Command getCommand(Scanner scanner) {
		String name = scanner.next();
		CommandFactory factory = commandFactoryByName.get(name);
		if (factory == null) {
			return new NotFoundCommand(name);
		} else {
			return factory.createCommand(scanner);
		}
	}
}
