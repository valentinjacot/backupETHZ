package patterns.command.gof;

import java.util.ArrayList;
import java.util.List;

public class Macro {
    private final List<Command> actions;
 
    public Macro() {
        actions = new ArrayList<>();
    }
 
    public void record(Command cmd) {
        actions.add(cmd);
    }
 
    public void run() {
        actions.forEach(Command::execute);
    }
}