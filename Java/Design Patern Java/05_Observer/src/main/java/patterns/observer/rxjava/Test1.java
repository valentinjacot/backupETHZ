package patterns.observer.rxjava;

import io.reactivex.Observable;
import io.reactivex.Observer;
import io.reactivex.disposables.Disposable;

public class Test1 {
	
	static String result = "";
	
	public static void main(String[] args) {
		Observable<String> observable = Observable.just("a", "b", "c", "d", "e", "f", "g");
		
		observable.subscribe(new Observer<String>() {
			@Override
			public void onSubscribe(Disposable d) {
				System.out.println("subscribed");
			}

			@Override
			public void onNext(String t) {
				System.out.println("onNext");
				result += t;
			}

			@Override
			public void onError(Throwable e) {
				System.out.println("onError");
			}

			@Override
			public void onComplete() {
				System.out.println("onComplete");
			}
		});
		
		System.out.println(result);
		
		result = "";
		observable.subscribe(
				i -> result += i, // OnNext
				Throwable::printStackTrace, // OnError
				() -> result += "_Completed" // OnCompleted
		);
		System.out.println(result);
	}

}
