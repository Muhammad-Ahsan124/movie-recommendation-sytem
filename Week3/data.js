// data.js
// Responsible for loading and parsing the MovieLens 100K u.item and u.data files.

// Globals (intentionally var to expose to window)
var items = {};           // map movieId -> { id, title, year?, genres? }
var ratings = [];         // array of { userId, itemId, rating }
var numUsers = 0;
var numMovies = 0;

// arrays used for training convenience
var userIdArray = [];
var itemIdArray = [];
var ratingValueArray = [];

/**
 * loadData()
 * Fetches u.item and u.data and parses them.
 * Returns a Promise that resolves when both files are parsed.
 */
async function loadData() {
  // MovieLens 100K raw file URLs
  const ITEM_URL = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.item';
  const DATA_URL = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.data';

  // Update simple status via DOM if available
  updateStatus && updateStatus('Fetching MovieLens files...');

  // Fetch both files in parallel
  const [itemResp, dataResp] = await Promise.all([
    fetch(ITEM_URL),
    fetch(DATA_URL)
  ]);

  if (!itemResp.ok || !dataResp.ok) {
    throw new Error('Failed to fetch dataset files. Check network or CORS.');
  }

  const [itemText, dataText] = await Promise.all([
    itemResp.text(),
    dataResp.text()
  ]);

  parseItemData(itemText);
  parseRatingData(dataText);

  // Build arrays for training (1-based ids preserved)
  userIdArray = ratings.map(r => r.userId);
  itemIdArray = ratings.map(r => r.itemId);
  ratingValueArray = ratings.map(r => r.rating);

  // Update derived counts
  numUsers = Math.max(...userIdArray);
  numMovies = Math.max(...itemIdArray);

  updateStatus && updateStatus(`Loaded ${Object.keys(items).length} movies and ${ratings.length} ratings. Found ${numUsers} users and ${numMovies} movies.`);
}

/**
 * parseItemData(text)
 * Parses u.item file format.
 * Each line is:
 * movie id | movie title | release date | video release date | IMDb URL | genres (19 fields)
 * We primarily extract movie id and title.
 */
function parseItemData(text) {
  items = {};
  const lines = text.split(/\r?\n/);
  for (const l of lines) {
    if (!l.trim()) continue;
    // The title may include '|' characters in some odd datasets; u.item uses '|' as delimiter.
    // Split into at most 5 + genres parts to be safe.
    const parts = l.split('|');
    const id = parseInt(parts[0], 10);
    const title = parts[1] ? parts[1].trim() : `Movie ${id}`;
    // store minimal info
    items[id] = {
      id: id,
      title: title
      // release_date, url, genres could be parsed if needed
    };
  }
}

/**
 * parseRatingData(text)
 * Parses u.data file format.
 * Each line:
 * user id \t item id \t rating \t timestamp
 */
function parseRatingData(text) {
  ratings = [];
  const lines = text.split(/\r?\n/);
  for (const l of lines) {
    if (!l.trim()) continue;
    const parts = l.split(/\s+/); // whitespace-separated
    if (parts.length < 3) continue;
    const userId = parseInt(parts[0], 10);
    const itemId = parseInt(parts[1], 10);
    const rating = parseFloat(parts[2]);
    if (Number.isNaN(userId) || Number.isNaN(itemId) || Number.isNaN(rating)) continue;
    ratings.push({ userId, itemId, rating });
  }
}
