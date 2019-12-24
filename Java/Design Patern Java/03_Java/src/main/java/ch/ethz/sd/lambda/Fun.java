package ch.ethz.sd.lambda;

@FunctionalInterface
public interface Fun<Input, Output> {
	Output apply(Input arg); // this is the single abstract method
}
