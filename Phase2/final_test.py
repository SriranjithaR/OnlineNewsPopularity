def finalTest(clf,x_test,y_test_binary):
    finalScore = clf.score(x_test,y_test_binary)
    print "Final score : ", finalScore
