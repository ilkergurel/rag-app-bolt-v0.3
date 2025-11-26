import { useContext } from 'react';
import { AuthContext } from './context/AuthContext';
import Auth from './components/Auth';
import Chat from './components/Chat';

function App() {
  const { user, loading } = useContext(AuthContext);

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-slate-50">
        <div className="text-slate-600 text-lg">Loading...</div>
      </div>
    );
  }

  return user ? <Chat /> : <Auth />;
}

export default App;
