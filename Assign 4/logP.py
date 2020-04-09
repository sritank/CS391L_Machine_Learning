def logP(data,theta):
    X = data[:,0].reshape(-1,1);
    X=X*1.0;

    # X = np.array([-2.1, -1.5, -0.7, 0.3, 1.0, 1.8, 2.5]).reshape(-1,1);

    Y = data[:,1].reshape(-1,1);
    sigma_f = theta[0];
    sigma_l=theta[1];
    sigma_n = theta[2];

    L = X.size

    k = np.zeros([L,L]);
    kl = np.zeros([L,L]);
    tol=1;
    count=0;

    Xii = np.multiply(XX,XX);
    Xii = np.matlib.repmat(Xii,1,L);
    Xjj = Xii.transpose();
    # ipdb.set_trace();
    Xi_Xj = Xii+Xjj-2*X.dot(X.transpose());


    Kp = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*Xi_Xj);

    Kl = -0.5*np.exp(sigma_l)*Xi_Xj;


    Q = Kp + np.eye(L)*np.exp(sigma_n);
    Lc=np.linalg.cholesky(Q);
    beta=np.linalg.solve(Lc.transpose(),np.linalg.solve(Lc,Y));
    Qinv = np.linalg.solve(Lc.transpose(),np.linalg.solve(Lc,np.eye(L)));
    # Qinv = LA.inv(Q);



    logPosterior = -0.5*Y.T.dot(beta) - 0.5*log(LA.det(Q)) - L/2*np.log(2*np.pi);

    print("sigma_l=",sigma_l)
    print("sigma_n=",sigma_n)
    print("sigma_f=",sigma_f)
    print("det=",LA.det(Q))

    return -logPosterior
    # ipdb.set_trace();
